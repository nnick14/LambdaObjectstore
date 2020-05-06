package dashboard

import (
	ui "github.com/gizak/termui/v3"
	"log"
	"time"

	"github.com/mason-leap-lab/infinicache/proxy/dashboard/views"
	"github.com/mason-leap-lab/infinicache/proxy/global"
)

type Dashboard struct {
	*ui.Grid
	clusterView   *views.ClusterView
	logView       *views.LogView
}

func NewDashboard() *Dashboard {
	if err := ui.Init(); err != nil {
		log.Panic("Failed to initialize dashboard: %v", err)
	}

	dashboard := &Dashboard{
		Grid: ui.NewGrid(),
		clusterView: views.NewClusterView(" Nodes "),
		logView: views.NewLogView(" Logs ", global.Options.LogFile),
	}

	// Full screen
	termWidth, termHeight := ui.TerminalDimensions()
	dashboard.Grid.SetRect(0, 0, termWidth, termHeight)

	// Layout
	dashboard.Grid.Set(
		ui.NewRow(1.0/3,
			ui.NewCol(1.0/1, dashboard.clusterView),
		),
		ui.NewRow(2.0/3,
			ui.NewCol(1.0/1, dashboard.logView),
		),
	)

	return dashboard
}

func (dash *Dashboard) Update() {
	ui.Render(dash)
}

func (dash *Dashboard) Start() {
	uiEvents := ui.PollEvents()
	ticker := time.NewTicker(time.Second).C
	for {
		dash.Update()
		select {
		case e := <-uiEvents:
			switch e.ID {
			case "q", "<C-c>":
				return
			case "<Resize>":
				payload := e.Payload.(ui.Resize)
				dash.SetRect(0, 0, payload.Width, payload.Height)
				ui.Clear()
				// ui.Render(dash)
			}
		case <-ticker:
			// ui.Render(dash)
		}
	}
}

func (dash *Dashboard) Close() {
	ui.Close()
}
